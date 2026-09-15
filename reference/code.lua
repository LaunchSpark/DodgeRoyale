-- dodge
-- a game by artridge

-- bug de passthrough
-- cap bouncecap
-- baisser bouncecap p checkered

poke(0x5f2d,1)

function _init()
 cartdata("artridge_dodge")
 menuitem(1,"reset highscores",function() for i=0,11 do dset(i,0) end iniths() end)
 --menuitem(2,"toggle music",function() m=not m end)
 --menuitem(3,"toggle sfx",function() s=not s end )
 -- main functions
 _upd,_drw=updatemenu,drawmenu

 -- objects
 part,enemies,debug={},{},{}

 -- player
 px,py,pspd,pf=64,64,0.5,0.8
 dirc=0
 score=0
 shouldcollide=true
 eshouldcollide=true

 -- enemies
 ed,es=0,3
 
 -- easy static:
 -- easy normal:
 -- normal static:
 -- normal moving:
 -- hard static: 
 -- hard moving: 0.45
-- bouncecaps=0.4
-- bouncecapm=0.4
-- minbounces=0.4
-- minbouncem=0.4
 initspawns()

 -- powerups
 size,sizetimer=4,0
 freeze,frztimer,frz=false,0,1

 -- patterns
 cp,cpt,pt=nil,0,7
 initpatterns()

 -- misc
 shake=0
 f=0.99
 input=0
 ftimer=0
 dpx,dpy=0,0
 trsy,trsdone,trspx,trspy=-128,true,0,0
 dirx,diry={-1,1,0,0},{0,0,-1,1}
 prevxpos,prevypos=0,0
 fy=true
 m,s=false,true

 -- settings
 smallsettings={{12,1},2,true,true}
 settings={
  {
         {
          current=defcur(dget(12)),
             name="theme",
             values={"blue","dark blue","green","indigo","purple","orange","pink","grey","dark grey","black","neon red","neon blue","neon green"},
          meanings={{12,1},{1,0},{3,1},{13,1},{2,1},{9,8},{14,2},{6,5},{5,0},{0,0},{0,8},{0,12},{0,11}}
         }
     },
     {
      {
       current=defcur(dget(13)),
             name="difficulty",
             values={"normal","hard","easy"},
          meanings={2,3,1}
      },
      {
       current=defcur(dget(14)),
             name="patterns",
             values={"on","off"},
          meanings={true,false}
      },
      {
       current=defcur(dget(15)),
             name="powerups",
             values={"on","off"},
          meanings={true,false}
      }
     }
 }
 menuitem(2,"save settings",function() dset(12,settings[1][1].current)
 for i=1,3 do
  dset(12+i,settings[2][i].current)
 end end)
 if dget(12)!=0 then
  settings[1][1].current=dget(12)
     for i=1,3 do
      settings[2][i].current=dget(12+i)
     end
 end
 
 iniths()
 catnames,mbtn,mt={"","gameplay"},{},0
 cursi,newhs,scanclick=1,false,true
 difspd,difest={0.002,0.0069,0.028},{0.005,0.01,0.02}
 startspd,startest={0.8,1,1.6},{1.4,1,0.8}
 startbs={0.8,0.65,0.6}
 incbs={0.0005,0.019,0.024}
 tarbs={0.45,0.45,0.4}
 startbm={1.05,0.9,0.6}
 incbm={0.0013,0.019,0.024}
 tarbm={0.45,0.45,0.4}
 updatesettings()
 --[[
 target and from :
 0 : menu
 1 : game
 2 : settings
 ]]

 -- music(20,150)
 music(3)
 sfx(55,2)
 
 -- up: settings
 -- down: logo and game
end

function _update60()

-- debug[1]=bouncecapm
-- debug[2]=bouncecaps
 
 -- update music
-- for i=0,24 do
--  set_speed(i,frz==1 and 19 or 30)
-- end
 
 -- set settings
 dopatterns=smallsettings[3]
 local powerups=smallsettings[4]
 local powerupv=powerups and 2 or 0
 enemiestats={76.5,17.5,powerupv,powerupv,powerupv}
 for e in all(enemies) do
  if e.p>=2 then
   enemiestats[e.p+1]-=2
   for i=3,5 do
    if enemiestats[i]!=0 then
     enemiestats[i]+=(e.p!=i-1 and 1 or 0)
    end
    enemiestats[i]=mid(0,enemiestats[i],3)
   end
  end
 end
 enemiestats[4]=score<=10 and 0 or enemiestats[4]
 
 _upd()

 -- switch input
 input=anybtn() and 1 or anymouse() and 0 or input
 
 -- shake
 shake=min(shake,0.1)
 local shakex,shakey=16-rnd(32),16-rnd(32)

 shakex*=shake
 shakey*=shake

 camera(shakex,shakey)

 shake*=0.95
 if shake<0.05 then
  shake=0
 end
end

function _draw()
 cls(12)
 _drw()
 pal(12,smallsettings[1][1],1)
 pal(1,smallsettings[1][2],1)

 -- debug
 cursor(4,8,8)
 for d in all(debug) do
  print(d)
 end
end

function iniths()
    highscores={}
    for i=1,12 do
     add(highscores,{dget(i-1),{ceil(i/4),i%4==1 or i%4==2,i%4==1 or i%4==0}})
    end
end

function initspawns()
 spawns={{-10,-10},{138,-10},{-10,138},{138,138}}
end
-->8
-- updates

function updatemenu()
 -- startgame
 if trsdone and (btnp(❎) or stat(34)==1) then
  _upd,_drw,target,from,hasplayed=updatetransition,drawtransition,1,0,true
  setrsy()
  sfx(55,-2)
 end
 
 -- tosettings
 if _upd==updatemenu and btnp(🅾️) or stat(34)==2 and trsdone then
  _upd,_drw,target,from=updatetransition,drawtransition,2,0
  sfx(55,-2)
  sfx(55,0)
  setrsy()
 end
end

function updatesettings()
 local gmply=(isdead or not hasplayed)
 if mt<=0 then 
  mbtn,mt={},6
 else
  mt-=1
 end
 backtomenu(2)
 if scanclick then
  local y,i2=0,0
  for i=1,#settings do
      local c,n=settings[i],catnames[i]
      y+=20
      for i=1,#c do
       local s=c[i]
       i2+=1
       y+=10
       --local value="< "..s.values[s.current].." >"
       if    (stat(34)==2 and
          70>stat(32) and
                         y+6>stat(33) and
                         62<stat(32) and
                         y<stat(33)) or
                         (btnp(⬅️) and cursi==i2)
             then
              if gmply or n!="gameplay" then
                        s.current-=1
                        sfx(58)
                        if (s.current<=0) s.current=#s.values
                        mbtn={96,64,y-1}
                    end       
                end
                if (stat(34)==2 and
                   118>stat(32) and
                         y+6>stat(33) and
                         110<stat(32) and
                         y<stat(33)) or
                         (btnp(➡️) and cursi==i2)
             then
              if gmply or n!="gameplay" then
                        s.current+=1
                        sfx(58)
                        if (s.current>#s.values) s.current=1
      mbtn={80,113,y-1}
                 end
                end
                smallsettings[i2]=s.meanings[s.current]
      end
     end
 end
 if gmply then
  local s=smallsettings[2]
  spd,est,bouncecaps,bouncecapm=startspd[s],startest[s],startbs[s],startbm[s]
 end
 scanclick=stat(34)!=2
 if (btnp(⬇️)) cursi+=1
 if (btnp(⬆️)) cursi-=1
 cursi=mid(1,cursi,gmply and 4 or 1)
end

function updategame()
 if (fy) updatefyou()
 backtomenu(1)
 
 -- restart
 if isdead and canclick and (stat(34)==1 or btnp(❎)) then
  reset()
  music(3)
  sfx(55,-2)
 end
 canclick=not (stat(34)==1 or btnp(❎))
 
 collisioncheck()
 
 -- movement
 if input==0 then
  px=stat(32)
  py=stat(33)
 else
  dpx*=pf
  dpy*=pf

        for i=1,4 do
         if btn(i-1) then
          dpx+=dirx[i]*pspd
          dpy+=diry[i]*pspd
         end
        end

  px+=dpx
  py+=dpy
 end
 px,py=mid(2,px,125),mid(2,py,125)
 
 updateparts()
 
 -- freeze handle
 if freeze then
  frztimer+=1
  if frztimer<=420 then
   frz=0.4
  else
   frz+=0.6/60
  end
  if frztimer>480 then
   freeze=false
   frz=1
  end
 end
 
 -- size handle
 if size!=4 then
  sizetimer+=frz
  if sizetimer>600 then
   if size<4 then
       size+=0.5*frz
   else
    size=4
   end
  end
 end
 
 if not isdead then
  spawnenemies()
  updateenemies()
  updatepatterns()
 end
end

function updatetransition()
   
end
-->8
-- draws

function drawmenu()
 spr(17,6,18,14,7)
 print2("press "..(input==1 and "❎" or ""),48,58)
 if (input==0) spr(192,71,56,1,2)

 print2("eddy rashed",42,96)
 print2("oskar zanota",40,103)
end

function drawsettings()
 spr(160,38,8,8,2)
 local y,i2=0,0
 for i=1,#settings do
  y+=20
  local c,n=settings[i],catnames[i]
  print2(n,64-#n*2,y)
  for i=1,#c do
   y+=10
   i2+=1
   if (input==1 and i2==cursi) spr(64,5,y)
   local s=c[i]
   print2(s.name,20,y)
   local value=s.values[s.current]
   print2(value,91-#value*2,y)
   if ((isdead or not hasplayed) or n!="gameplay") print2("<           >",65,y)
  end
 end

 -- highscores
 if geths()>0 then
     local txt="highscore: "..ceil(geths())
     print2(txt,64-#txt*2,95)
    else
     print2("no highscore",40,95)
    end
   
    if (#mbtn!=0) spr(mbtn[1],mbtn[2],mbtn[3])

 print2((input==1 and "❎" or "  ").." to exit",1,110)
 if (input==0) spr(192,2,108,1,2)
 print2((input==1 and "⬅️➡️" or "  ").." to change a setting",1,120)
 if (input==0) spr(193,2,118,1,2)
 if (input==0) spr(1,stat(32),stat(33))
end

function drawgame()
 if not isdead then
 
  ------- particles -------
  -- shadows
     for _p in all(part) do
      if _p.tpe==0 then
       circfill(_p.x,_p.y+1,_p.r,1)
      else
       pset(_p.x,_p.y+1,1)
       pset(_p.x,_p.y,_p.col)
      end
     end
   
     -- actual particle
     for _p in all(part) do
      if _p.tpe==0 then
       circfill(_p.x,_p.y,_p.r,7)
      end
     end
     
     
  ------ enemies -------
  for _e in all(enemies) do
      local x,y,s=flr(_e.x),flr(_e.y),flr(_e.s)
      if _e.p==2 then
       cols={8,9,10}
      elseif _e.p==3 then
       cols={1,13,7}
      elseif _e.p==4 then
       cols={7}
      end
   
      if _e.p==0 then
       rect2(x,y,x+s,y+s)
      elseif _e.p==1 or _e.p==-1 then
       rect2(x,y,x+s,y+s,true)
      elseif _e.p>=2 then
       circfill(x,y+1,4,1)
       spawntrail(x,y,cols)
       circfill(x,y,4,7)
      end
     end
 
  ------- patterns --------
  if cp then
      for r in all(cp.rects) do
       local x,y,w,h=round(r.x),round(r.y),round(r.w),round(r.h)
       local xw,yh=x+w,y+h
       if r.sh<2 and cp.tpe==0 then
        local minus,dot=0,true
        if r.sh>1 then
         fillp_dot(x,y)
         rect(x,y+1,xw,y+h+1,1)
         fillp_dot(x,y+1)
         rect(x,y,xw,y+h,7)
         fillp()
         minus=1
         dot=false
        end
         line2(x,y,lerp(x,x+w/2,r.sh-minus),y,dot)
         line2(xw,y,lerp(xw,x+w/2,r.sh-minus),y,dot)
         line2(xw,y,x+w,lerp(y,y+h/2,r.sh-minus),dot)
         line2(xw,yh,xw,lerp(yh,y+h/2,r.sh-minus),dot)
         line2(x,yh,lerp(x,x+w/2,r.sh-minus),y+h,dot)
         line2(xw,yh,lerp(xw,x+w/2,r.sh-minus),y+h,dot)
         line2(x,y,x,lerp(y,y+h/2,r.sh-minus),dot)
         line2(x,yh,x,lerp(yh,y+h/2,r.sh-minus),dot)
       else
        if cp.timer<125 then
         for l in all(r.warnings) do
          line2(l[1],l[2],l[3],l[4],false)
         end
        end
        rectfill(x,y+1,xw,yh+1,1)
        rectfill(x,y,xw,yh,7)
       end
      end
     end
     
  rectfill(0,0,#tostr(ceil(score))*4,7,12)
  print2(ceil(score),1,1)
  addpart(px,py,0,0,size,0,10,{7})
 else
  print2("press "..(input==1 and "🅾️" or "  ").." to open settings",14,110)
  if (input==0) spr(192,44,118,1,2)
  print2("press "..(input==1 and "❎" or "  ").." to play again",20,120)
  if (input==0) spr(193,38,108,1,2)
  spr(128,34,40,8,2)
  local str="score:"..ceil(score)
  print2(str,64-#str*2,56)
  if (newhs) wave()
 end
end

function drawtransition()
 local updstates={updatemenu,updategame,updatesettings}
 local drwstates={drawmenu,drawgame,drawsettings}
 
 
 if trsy<=0 then
     st=target==2 and 3 or from+1
    else
  st=target==2 and from+1 or target+1
 end
 
 updstates[st]()
 drwstates[st]()
 trsy+=(target==2 and -10 or 10)
 if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then
  trsdone=true
  _upd,_drw=updstates[target+1],drwstates[target+1]
 else
  trsdone=false
 end
 rectfill(0,trsy,127,trsy+127,7)
end

-->8
-- gameplay

function collisioncheck()
 if not isdead and shouldcollide then
  for _e in all(enemies) do
   if _e then
       if _e.p>=2 then
        if    px+size > _e.x-4 and
                    py+size > _e.y-4 and
                    px-size < _e.x+4 and
                    py-size < _e.y+4
        then
          collide(_e)
           end
    else
        if    px+size-1 > _e.x and
                 py+size-1 > _e.y and
                 px-size+1 < _e.x+_e.s and
                 py-size+1 < _e.y+_e.s
        then
         collide(_e)
     end
    end
   end
     end

     if cp then
         for r in all(cp.rects) do
          if r.sh==2 or cp.tpe==1 then
              if    px-2+size > r.x and
                     py-2+size > r.y and
                     px+2-size < r.x+r.w-1 and
                     py+2-size < r.y+r.h-1
           then
            die()
           end
          end
         end
     end
 end
end

function collide(_e)
 shatter(_e.x,_e.y)
 shake+=0.07
 if _e.p<2 then
  die()
 elseif _e.p==3 then
  freeze=true
  frztimer=0
  score+=1
  difficultycurve()
  sfx(62)
 elseif _e.p==4 then
  size=2
  sizetimer=0
  sfx(61)
  score+=1
  difficultycurve()
 elseif _e.p==2 then
  sfx(60)
     for e in all(enemies) do
      if e.p==1 then
       kamikaze(e)
      elseif e.p!=-1 then
       shatter(e.x,e.y)
      end
   if (e.p!=-1) del(enemies,e)
   score+=1
   difficultycurve()
  end
 end
 del(enemies,_e)
end

function reset()
 -- _init() ?
 isdead,score,shake=false,0,0
 ed,size=0,4
 local s=smallsettings[2]
 spd,est,bouncecaps=startspd[s],startest[s],startbs[s]
 bouncecapm=startbm[s]
    sizetimer,freeze=0,false
 frztimer,frz=0,1
 enemies,part={},{}
 newhs,cp,cpt,pt,fy=false,nil,0,7,true
 initspawns()
 bouncecap=1.45
 local counters={}
 for i,p in pairs(patterns) do
  add(counters,{i,p.counter})
 end
 initpatterns()
 for i in all(counters) do
  local ix=i[1]
  patterns[ix].counter=i[2]
  local p=patterns[ix]
  local pr=p.prob
  patterns[ix].prob=(p.tpe==1 and 17 or 15)/flr(#p.var+1)-p.counter
  if pr==1 then
   patterns[ix].prob-=6*patterns[ix].counter
  end
  patterns[ix].prob+=2
 end
end

function backtomenu(_from)
 if trsdone then
  if _from==1 then
   if btnp(🅾️) or stat(34)==2 then
    _upd,_drw=updatetransition,drawtransition
    target,from=2,_from
    if (isdead) music(3)
    sfx(55,-2)
    sfx(55,0)
   end
  else
   if btnp(❎) or stat(34)==1 then
    _upd,_drw=updatetransition,drawtransition
    target,from=hasplayed and 1 or 0,_from
    if hasplayed then
     if (isdead) music(22)
     sfx(55,-2)
    else 
     sfx(55,2)
    end
   end
  end
  setrsy()
 end
end

function difficultycurve(half)
 local s,h=smallsettings[2],half and 2 or 1
 
 local spdinc,estinc=difspd[s]/h,difest[s]/h
 spd,est=lerp(spd,3,spdinc),lerp(est,0.22,estinc)

 local bincs,btars=incbs[s]/h,tarbs[s]
 bouncecaps=lerp(bouncecaps,btars,bincs)

 local bincm,btarm=incbm[s]/h,tarbm[s]
 bouncecapm=lerp(bouncecapm,btarm,bincm)
end

function spawnenemies()
 ed+=1
 local spp=cp and (cp.special==2 and 1.75 or cp.special==3 and 1.75) or 1 
 if ed>est*60*spp*(2-frz) and #spawns>0 then
  ed=0
  local x,y,d
  
  -- enemy spawn corner
  for s in all(spawns) do
   if not d or dist(s[1],s[2],px,py)<d then
                d,x,y=dist(s[1],s[2],px,py),s[1],s[2]
   end
  end

  -- enemy size
  local esc=rnd(100)
  if esc<=20 then es=3 end
  if esc>20 and esc<=70 then es=4 end
  if esc>70 and esc<=90 then es=5 end
  if esc>90 and esc<=100 then es=6 end

  -- enemy personnality
  -- -1 kamikaze explosion rectangle
  -- 0 normal
  -- 1 kamikaze
  -- 2 explosion powerup
  -- 3 freeze powerup
  -- 4 size powerup
  -- 75 90 91 92
  
  local pers={}
  for k,p in pairs(enemiestats) do
   for i=1,p do
    add(pers,k-1)
   end
  end
  ep=pers[flr(rnd(#pers))+1]

  -- finally spawn the enemy
  addenemy(x,y,false,1,es,ep)
 end
end

function updatefyou()
 if    px+size-2>60 and
         py+size-2>60 and
         px-size+2<68 and
            py-size+2<68
 then
  ftimer+=1
  if ftimer>50 then
   ftimer=0
      fyou()
  end
 else
  ftimer=0
 end
end

-- screen corners
function fyou2()
 for x=0,1 do
     for y=0,1 do
            addenemy(x*128,y*128,false,1,es,0)
     end
    end
end

-- all spawn points
function fyou()
 for s in all(spawns) do
  addenemy(s[1],s[2],false,1,es,0)
 end
end

function checkhs()
 if score>geths() then
  for i,h in pairs(highscores) do
   if iscurrenths(h) then
    h[1]=ceil(score)
    dset(i-1,score)
    newhs=true
   end
  end
 end
end

function geths()
 for h in all(highscores) do
  if iscurrenths(h) then
   return h[1]
  end
 end
end

function iscurrenths(h)
 return h[2][1]==smallsettings[2]
 and h[2][2]==smallsettings[3]
 and h[2][3]==smallsettings[4]
end

function die()
 sfx(62)
 isdead=true
 checkhs()
 shake+=0.07
 music(22)
 --wait(30)
end

function setrsy()
 trsy=target==2 and 128 or -128
end
-->8
-- juicy
function addpart(_x,_y,_dx,_dy,_r,_type,_maxage,_col)
 add(part,{
 x=_x,
 y=_y,
 dx=_dx,
 dy=_dy,
 r=_r,
 tpe=_type,
 mage=_maxage,
 age=0,
 col=0,
 colarr=_col})

 --[[
  0 : player trail
  1 : explosion parts+
      powerups trail
 ]]
end

function updateparts()
 for _p in all(part) do
  _p.age+=1
  if _p.age>_p.mage then
   del(part,_p)
  else
   -- change colors
   if #_p.colarr==1 then
    _p.col = _p.colarr[1]
   else
    local _ci=_p.age/_p.mage
    _ci=1+flr(_ci*#_p.colarr)
    _p.col = _p.colarr[_ci]
   end

   -- move particle
   _p.x+=_p.dx
   _p.y+=_p.dy

   if _p.tpe!=2 then
   _p.r*=0.90
   end
   if _p.r<0 and _p.tpe==0 then
    del(part,_p)
   end
  end
 end
end

function shatter(_x,_y)
 for i=0,10 do
  local _ang = rnd()
  local _dx = sin(_ang)*1
  local _dy = cos(_ang)*1

  addpart(_x,_y,_dx,_dy,0,1,60,{7})
 end
end

function spawntrail(_x,_y,_cols)
 local _ang=rnd()
 local _ox=sin(_ang)*2.4
 local _oy=cos(_ang)*2.4
 addpart(_x+_ox,_y+_oy,0,0,0,1,20+rnd(15),_cols)
end
-->8
-- enemies
function addenemy(_x,_y,_isdying,_s,_ms,_ep)
 add(enemies,{
  x=_x,
  y=_y,
  vx=0,
  vy=0,
  maxs=_ms,
  f=0,
  s=_s,
  p=_ep,
  spd=1,
  isizing=true,
  life=_ep==-1 and 60
 })
end

function updateenemies()
 for _e in all(enemies) do
  local x,y,vx,vy,s=_e.x,_e.y,_e.vx,_e.vy,_e.s
     if s<_e.maxs and _e.p!=-1 then
      _e.s+=frz
     end
     if s<_e.maxs and _e.p==-1 and _e.isizing then
      _e.s+=frz
   _e.x-=0.5*frz
   _e.y-=0.5*frz
     end
     if s>=_e.maxs and _e.p==-1 then
      _e.isizing=false
     end
     if _e.p==-1 and _e.isizing==false then
      _e.s-=frz
   _e.x+=0.5*frz
   _e.y+=0.5*frz
     end
     if (_e.p>=2) _e.s=4
         if _e.life then
          _e.life-=frz
          if _e.life<=0 then
           del(enemies,_e)
          end
         end
         if mid(1,x,128)==x and mid(0,y,128) then
          _e.ins=true
         end
         if _e.isdying then
    if _e.p!=1 then
                 shatter(x,y)
             else
              kamikaze(_e)
             end
          del(enemies,_e)
          sfx(63)
          score+=0.5
          difficultycurve(true)
             shake+=0.07
            end
   -- big section about movement
         if _e then
          vx*=f
          vy*=f
          if _e.p>=0 then
           if x<px then
            vx+=0.01
           elseif x>px then
            vx-=0.01
           end
           
           if y<py then
            vy+=0.01
           elseif y>py then
            vy-=0.01
           end

          end

                -- pattern collision
                local pw=_e.p>1
    if cp and _e.p!=-1 then
     for r in all(cp.rects) do
      if r.sh==2 then
                         local dx,dy=(r.dx and r.dx or 0),(r.dy and r.dy or 0)
          if x+vx-(pw and 3 or 0)<r.x+r.w and
                      x+s>r.x+dx and
                      y+s>r.y+dy and
                      y+vy-(pw and 3 or 0)<r.y+r.h then
           local minbounce=cp.tpe==0 and bouncecaps or bouncecapm
           if x+vx<r.x then
            vx=-max(abs(vx),minbounce)
           elseif x+vx+s>r.x+r.w then
            vx=max(abs(vx),minbounce)
           end
           if y+vy<r.y then
            vy=-max(abs(vy),minbounce)
           elseif y+vy+s>r.y+r.h then
            vy=max(abs(vy),minbounce)
           end
           -- cap bounce
        local bouncecap=cp.tpe==0 and bouncecaps or bouncecapm
                                if (cp.bouncecap) bouncecap-=0.15
                               
                               
                                vx=min(abs(vx),bouncecap)*sgn(vx)
                                vy=min(abs(vy),bouncecap)*sgn(vy)
          end
         end
     end
    end
               
                -- die if crushed by pattern
    if cp and _e.p!=-1 then
     for r in all(cp.rects) do
      if r.sh==2 then
                         local should,addx,addy=false,0,0
                   if x+s>127 then
                                should=true
                                addx=-4
                            elseif x-s<0 then
                             should=true
                                addx=4
                            end
   
                            if y+s>127 then
                                should=true
                                addy=-4
                            elseif y-s<0 then
                                should=true
                                addy=4
                            end

--                                _e.isdying=x+addx > r.x and
--                                            y+addy > r.y and
--                                            x+addx < r.x+r.w and
--                                            y+addy < r.y+r.h and
--           should and _e.ins
                            if should and _e.ins then
                                if    x+addx > r.x and
                                            y+addy > r.y and
                                            x+addx < r.x+r.w and
                                            y+addy < r.y+r.h
                                then
                                _e.isdying=true
                                end
                            end
                        end
     end
    end
    
    -- bounce off edges
    if _e.ins and _e.p!=-1 then
                    if y>=129-s then
                        vy=-abs(vy)
                    elseif y<=-2 then
                        vy=abs(vy)
                    end

                    if x>=129-s then
                        vx=-abs(vx)
                    elseif x<=-2 then
                        vx=abs(vx)
                    end
                end
               
    -- collision
    for _oe in all(enemies) do
           local x,y,ox,oy,s,os=_e.x,_e.y,_oe.x,_oe.y,_e.s,_oe.s
           if _e!=_oe and _e.ins and _oe.ins and eshouldcollide then
            if _e.p!=-1 and _oe.p!=-1 and _e and _oe then
             local sze,osze=_e.p>=2 and 4 or 0,_oe.p>=2 and 4 or 0
             if    x+s > ox-osze and
                      y+s > oy-osze and
                      x-sze < ox+os and
                      y-sze < oy+os
                then
                    _e.isdying,_oe.isdying=true,true
                end
               end
           end
          end

    -- kamikaze spd
    if _e.p==1 then
     local d=dist(px,py,_e.x,_e.y)
     if d<=25 then
      _e.spd-=0.01
     else
      if _e.spd<1 then
       _e.spd+=0.01
      else
       _e.spd=1
      end
     end
    end
       _e.x+=vx*spd*frz*_e.spd
       _e.y+=vy*spd*frz*_e.spd
       _e.vx=vx
       _e.vy=vy
   end
  end
end

function kamikaze(_e)
 addenemy(_e.x+_e.s/2,_e.y+_e.s/2,false,0,30,-1,0)
end

-->8
-- tools
function anybtn()
 for i=0,5 do
  if btn(i) then
   return true
  end
 end
end

function anymouse()
 mousemov=not (prevxpos==stat(32) and prevypos==stat(33))
 prevxpos,prevypos=stat(32),stat(33)
 return mousemov
 or stat(34)!=0
end

function print2(_txt,_x,_y)
 ?_txt,_x,_y+1,1
 ?_txt,_x,_y,7
end

function round(val)
 return val%1<0.5 and flr(val) or ceil(val)
end

function dist(x1,y1,x2,y2)
 return sqrt(((x1-x2)/16)^2+((y1-y2)/16)^2)*16
end

function fillp_dot(x,y)
 fillp(0b1010010110100101.1)
 if (x and x%2==y%2) fillp(0b101101001011010.1)
end

function lerp(pos,tar,perc)
 return (1-perc)*pos+perc*tar
end

function line2(x1,y1,x2,y2,f)
 local f2=f==nil and true or f
 if (f2) fillp_dot(x1,y1+1)
 line(x1,y1+1,x2,y2+1,1)
 if (f2) fillp_dot(x1,y1)
 line(x1,y1,x2,y2,7)
 fillp()
end

function range(val1,val2)
 return rnd(val2-val1)+val1
end

--function wait(a) for i = 1,a do flip() end end

function set_speed(s,spd)
 poke(0x3200+68*s+65,spd)
end

function wave(txt)
 local y
 local c
 local x=64-(#"new highscore!"*4)/2
 local f=time()*30
 for c=1,#"new highscore!" do
     y = sin((x+f)/25)*2
     print2(sub("new highscore!",c,c),x,(82-4)+y+1,1)
     x=x+4
 end
end

function rect2(x1,y1,x2,y2,f)
 local func=f and rect or rectfill
 func(x1,y1+1,x2,y2+1,1)
 func(x1,y1,x2,y2,7)
end

function defcur(val)
 return val==0 and 1 or val
end
-->8
-- patterns
-- without patterns : 3347
-- with one pattern : 3796 (+449)

function initpatterns()
 --[[
  draw types (drw):
   c = checked
   l = line
   f = filled

  movement type (mt):
      true = smooth
      false = linear

  checked fillp:
      fillp(0b0101101001011010.1)
 ]]

 patterns={
 {
   -- 1a 1
   var={2},
   rects={
    {
     x=28,
     y=16,
     w=20,
     h=20
    },
    {
     x=80,
     y=92,
     w=20,
     h=20
    }
   }
  },
  {
   -- 1b 2
   var={1},
   rects={
    {
     x=28,
     y=92,
     w=20,
     h=20
    },
    {
     x=80,
     y=16,
     w=20,
     h=20
    }
   }
  },
  {
   -- 2a 3
   mins=10,
   maxs=60,
   var={4},
   autovar=4,
   rects={
    {
     x=56,
     y=0,
     h=42
    },
    {
     x=56,
     y=85,
     h=42
    }
   }
  },
  {
   -- 2b 4
   var={3},
   rects={}
  },
  {
   -- 3aa 5 left
   var={6,7,8},
   autovar=6,
   tpe=1,
   rects={
    {
     x=131,
     y=32,
     w=24,
     h=64,
     spd=0.7,
     dx=-1,
     targets={{-26,32,24,64}}
    }
   }
  },
  {
   -- 3ab 6 up
   var={5,7,8},
   tpe=1,
   rects={}
  },
  {
   -- 3ba 7 down
   var={8,5,6},
   autovar=8,
   tpe=1,
   rects={
    {
     x=32,
     y=-28,
     w=64,
     h=24,
     spd=0.7,
     dy=1,
     targets={{32,130,64,24}}
    }
   }
  },
  {
   -- 3bb 8 right
   var={7,6,5},
   tpe=1,
   rects={}
  },
  {
   -- 4aa 9
   var={10,11,12},
   autovar=10,
   tpe=1,
   rects={
    {
     x=0,
     y=131,
     w=63,
     spd=0.7,
     dy=-1,
     targets={{0,-20,63,16}}
    },
    {
     x=64,
     y=-20,
     w=63,
     spd=0.7,
     dy=1,
     targets={{64,144,63,16}}
    }
   }
  },
  {
   -- 4ab 10
   var={9,11,12},
   tpe=1,
   rects={}
  },
  {
   -- 4ba 11
   var={9,10,12},
   autovar=12,
   tpe=1,
   rects={
    {
     x=131,
     y=0,
     h=63,
     spd=0.7,
     dx=-1,
     targets={{-18,0,16,63}}
    },
    {
     x=-20,
     y=64,
     h=63,
     spd=0.7,
     dx=1,
     targets={{144,64,16,63}}
    }
   }
  },
  {
   -- 4bb 12
   var={11,10,9},
   tpe=1,
   rects={}
  },
  {
   -- 5a 13
   mins=50,
   maxs=170,
   prob=19,
   var={14},
   spawns=true,
   rects={
    {
     x=0,
     y=0,
     w=127,
     h=32,
     targets={"fy",false,"s",{},"s",{{-10,62},{138,62}},8.5,"s",{{-10,-10},{138,-10},{-10,138},{138,138}}}
    },
    {
     x=0,
     y=96,
     w=127,
     h=31
    }
   }
  },
  {
   -- 5b 14
   mins=50,
   maxs=120,
   prob=19,
   var={13},
   spawns=true,
   rects={
    {
     x=0,
     y=0,
     w=32,
     h=127,
     targets={"fy",false,"s",{},"s",{{62,-10},{62,138}},8.5,"s",{{-10,-10},{138,-10},{-10,138},{138,138}}}
    },
    {
     x=96,
     y=0,
     w=31,
     h=127
    }
   }
  },
  {
  -- 6a 15
   mins=100,
   maxs=200,
   special=1,
   var={16},
   rects={}
  },
  {
  -- 6b 16
   mins=100,
   maxs=200,
   special=1.1,
   var={15},
   rects={}
  },
  {
  -- 7 17
   mins=25,
   maxs=90,
   mt=true,
   rects={
    {
     x=56,
     y=56,
     spd=34,
     targets={3,{40,40,48,48},3,{56,56,16,16},3}
    }
   }
  },
  {
  -- 8 18
   maxs=60,
   special=5,
   bouncecap=true,
   rects={}
  },
  {
  -- 9 19
   mins=60,
   maxs=120,
   special=2,
   bouncecap=true,
   spawns=true,
   rects={}
  },
  {
  -- 10 20
   mins=60,
   maxs=120,
   special=3,
   spawns=true,
   bouncecap=true,
   rects={}
  },
  {
   -- 11a 21
   mins=17,
   maxs=60,
   bouncecap=true,
   rects={
    {
     x=20,
     y=62,
     w=26,
     h=4
    },
    {
     x=81,
     y=62,
     w=26,
     h=4
    },
    {
     x=62,
     y=20,
     w=4,
     h=26
    },
    {
     x=62,
     y=81,
     w=4,
     h=26
    }
   }
  },
  {
   -- 11b 22
   mins=60,
   maxs=120,
   rects={
    {
     x=19,
     y=62,
     w=90,
     h=4
    },
    {
     x=62,
     y=19,
     w=4,
     h=90
    }
   }
  },
  {
  --12 23
   mins=120,
   tpe=1,
   rects={
    {
     x=-50,
     y=-50,
     w=0,
     h=0,
     targets={"s",{{-10,-10},{138,-10},{-10,138},{138,138},{64,-10},{138,64},{64,138},{-10,64}},10}
    }
   }
  },
  {
   -- 13 24
   mins=120,
   special=6,
   rects={}
  },
  {
   -- 14a 25 autoroute
   mins=40,
   maxs=100,
   var={26},
   spawns=true,
   rects={
    {
     x=0,
     y=0,
     w=127,
     h=20,
     targets={"fy",false,"s",{{138,83},{-10,83},{-10,38},{138,38}},8.5}
    },
    {
     x=0,
     y=107,
     w=127,
     h=20
    },
    {
     x=24,
     y=62,
     w=30,
     h=4
    },
    {
     x=74,
     y=62,
     w=30,
     h=4
    }
   }
  },
  {
   -- 14b 26 autoroute
   mins=40,
   maxs=100,
   var={25},
   spawns=true,
   rects={
    {
     x=0,
     y=0,
     w=20,
     h=127,
     targets={"fy",false,"s",{{83,128},{83,-10},{38,-10},{38,138}},8.5}
    },
    {
     x=107,
     y=0,
     w=20,
     h=127
    },
    {
     x=62,
     y=24,
     w=4,
     h=30
    },
    {
     x=62,
     y=74,
     w=4,
     h=30
    }
   }
  },
  {
   -- 15 27
   mins=10,
   maxs=60,
   special=7,
   rects={}
  },
  {
   -- 16 28
   maxs=60,
   rects={
    {
     x=32,
     y=0,
     w=63,
    },
    {
     x=32,
     y=111,
     w=63,
    },
    {
     x=0,
     y=32,
     h=63
    },
    {
     x=111,
     y=32,
     h=63
    }
   }
  },
  {
   -- 17 29
   mins=50, 
   maxs=100,
   rects={
    {
     x=0,
     y=62,
     w=32,
     h=4
    },
    {
     x=95,
     y=62,
     w=32,
     h=4
    },
    {
     x=62,
     y=0,
     w=4,
     h=32
    },
    {
     x=62,
     y=95,
     w=4,
     h=32
    }
   }
  },
  {
   -- 18aa 30 down
      tpe=1,
   var={31,32,33},
   autovar=31,
   rects={
    {
     x=0,
     y=-20,
     w=32,
     spd=0.7,
     dy=1,
     targets={{0,128,40,16}}
    },
    {
     x=87,
     y=-20,
     w=40,
     spd=0.7,
     dy=1,
     targets={{87,128,40,16}}
    }
   }
  },
  {
   -- 18ab 31 right
      tpe=1,
   var={30,32,33},
   rects={}
  },
  {
   -- 18ba 32 left
      tpe=1,
   var={30,31,33},
   autovar=33,
   rects={
    {
     x=130,
     y=0,
     h=40,
     spd=0.7,
     dx=-1,
     targets={{-20,0,16,40}}
    },
    {
     x=130,
     y=87,
     h=40,
     spd=0.7,
     dx=-1,
     targets={{-20,87,16,40}}
    }
   }
  },
  {
   -- 18bb 33 up
      tpe=1,
   var={30,31,32},
   rects={}
  },
--  {
--   -- 19aa 34 top right
--   mins=120,
--      var={35,36,37},
--      rects={
--          {
--              x=48,
--              y=95
--          },
--          {
--           x=16,
--              y=63
--          },
--          {
--           x=84,
--              y=0,
--              w=43,
--              h=43,
--              targets={"s",{{-10,-10},{138,-10},{-10,138}},8.5}
--          }
--      }
--  },
--  {
--   -- 19aa 35 bottom left
--   mins=120,
--      var={34,36,37},
--      rects={
--          {
--              x=64,
--              y=16
--          },
--          {
--           x=96,
--              y=48
--          },
--          {
--           x=0,
--              y=84,
--              w=43,
--              h=43,
--              targets={"s",{{-10,-10},{138,-10},{-10,138}},8.5}
--          }
--      }
--  },
--  {
--   -- 19aa 36 top left
--      var={34,35,37},
--      rects={
--          {
--              x=48,
--              y=95
--          },
--          {
--           x=16,
--              y=63
--          },
--          {
--           x=84,
--              y=0,
--              w=43,
--              h=43,
--              targets={"s",{{-10,-10},{138,-10},{-10,138}},8.5}
--          }
--      }
--  },
  {
   -- 19aa 34 bottom right
   mins=120,
      var={34,35,36},
      rects={
          {
              x=48,
              y=16
          },
          {
           x=16,
              y=48
          },
          {
           x=84,
              y=84,
              w=43,
              h=43,
              targets={"s",{{-10,-10},{138,-10},{-10,138}},8.5}
          }
      }
  },
  {
   -- 20 35
   mins=80,
   maxs=160,
   rects={
    {
     x=20,
     y=48,
     w=8,
     h=32
    },
    {
     x=48,
     y=20,
     w=32,
     h=8
    },
    {
     x=48,
     y=100,
     w=32,
     h=8
    },
    {
     x=100,
     y=48,
     w=8,
     h=32
    }
   }
  },
  {
   -- 21aa 36 down
   mins=20,
   maxs=90,
   autovar=37,
   var={37,38,39},
   rects={
       {
        x=54,
        y=53,
        w=20,
        h=74
       }
   }
  },
  {
   -- 21ab 37 right
   var={36,38,39},
   rects={}
  },
  {
   -- 21ba 38 left
   mins=20,
   maxs=90,
   autovar=39,
   var={36,37,39},
   rects={
       {
        x=0,
        y=54,
        w=74,
        h=20
       }
   }
  },
  {
   -- 21bb 39 up
   var={36,37,38},
   rects={}
  },
--  {
--   -- 22a 40
--      autovar=41,
--      var={41},
--      rects={
--       {
--        x=36,
--        y=34,
--        w=8,
--        h=60,
--    },
--    {
--     x=84,
--        y=34,
--        w=8,
--        h=60,
--    }
--      }
--  },
--  {
--   -- 22b 41
--   var={40},
--   rects={}
--  }
 }
 
 -- special patterns (requiering programatical generation)
 local spw={{44,-10},{83,-10},{-10,44},{-10,83},{44,138},{83,138},{138,44},{138,83}}
 for p in all(patterns) do

  local sp=p.special
  if sp==1 or sp==1.1 then
   for i=-8,120,32 do
    add(p.rects,{
     x=sp==1 and i or 60,
     y=sp==1 and 61 or i,
     w=sp==1 and 16 or 8,
     h=sp==1 and 6 or 16,
     targets={"fy",false,8.5}})
   end
  elseif sp==2 then
   for _x=16,96,40 do
    for _y=16,96,40 do
        add(p.rects,{
         x=_x,
         y=_y,
         w=15,
         h=15,
         spd=1,
         targets={"fy",false,"s",spw,8.5}})
    end
   end
  elseif sp==3 then
   for _x=16,106,45 do
    for _y=16,106,45 do
     add(p.rects,{
         x=_x,
         y=_y,
         w=8,
         h=8,
         targets={"fy",false,"s",spw,8.5}})
    end
   end
   for _x=39,84,45 do
    for _y=39,84,45 do
     add(p.rects,{
         x=_x,
         y=_y,
         w=8,
         h=8,
         spd=1})
    end
   end
  elseif sp==5 then
   for _x=0,1 do
    for _y=0,1 do
     add(p.rects,{
      x=_x*75,
      y=_y*111,
      w=52,
      targets={"s",{{64,-4},{129,64},{64,129},{-4,64}},8.5}
     })
    end
   end
   for _x=0,1 do
    for _y=0,1 do
     add(p.rects,{
      x=_x*111,
      y=_y*75,
      h=52
     })
    end
   end
   for _x=50,75,25 do
    for _y=-9,129,138 do
     add(p.rects,{
      x=_x,
      y=_y,
      w=2,
      h=7
     })
    end
   end
   for _y=50,75,25 do
    for _x=-9,129,138 do
     add(p.rects,{
      x=_x,
      y=_y,
      w=7,
      h=2
     })
    end
   end
  elseif sp==6 then
   local loc={
      {range(16,32),
       range(16,32),
       range(14,24),
       range(14,24)},
      {range(72,95),
       range(16,36),
       range(10,20),
       range(10,20)},
      {range(26,86),
       range(72,82),
       range(14,19),
       range(10,18)}
      }
   for i=1,3 do
    add(p.rects,{
     x=loc[i][1],
     y=loc[i][2],
     w=loc[i][3],
     h=loc[i][4]
    })
   end
  elseif sp==7 then
   local loc={
   {range(14,32),
       range(20,32),
       range(20,25),
       range(25,30)},
      {range(50,90),
       range(76,90),
       range(10,20),
       range(10,20)}}
      if rnd()>0.5 then
       loc[2]={
    range(66,90),
       range(50,90),
       range(10,20),
       range(10,20)}
      end
      for i=1,2 do
        add(p.rects,{
     x=loc[i][1],
     y=loc[i][2],
     w=loc[i][3],
     h=loc[i][4]
    })
      end
  end
  
  --[[ default values:
      - pattern prob (15)
      - pattern movement type (false)
      - pattern variants (none)
      - pattern type (0)

      - rect spd (12)
      - rect action index (1)
      - rect wait counter (0)
      - rect visibility target (true)
      - rect visibility counter (0)
  ]]
  p.mt=p.mt or false
  p.tpe=p.tpe or 0
  p.var=p.var or {}
  p.prob=p.prob or (p.tpe==1 and 17 or 15)/flr(#p.var+1)
  p.counter=0
  p.timer=0
  p.mins=p.mins or 0
  p.maxs=p.maxs or 32767
  for r in all(p.rects) do
   r.spd=r.spd or 12
   r.i=1
   r.wait=0
   r.shown=true
   r.sh=0
   r.w=r.w or 16
   r.h=r.h or 16
   r.targets=r.targets or {8.5}
   r.dx=r.dx or 0
   r.dy=r.dy or 0
  end
  
  -- automatic variants
  if p.autovar!=nil then
   for r in all(p.rects) do
        local _r={
         x=r.y,
         y=r.x,
         w=r.h,
         h=r.w,
         spd=r.spd,
         targets={}
        }
    if r.dx!=0 then
     _r.dy=r.dx
     _r.dx=0
    end
    if r.dy!=0 then
     _r.dx=r.dy
     _r.dy=0
    end
    for i=1,#r.targets do
     t=r.targets[i]
     if type(t)=="table" then
      add(_r.targets,{t[2],t[1],t[4],t[3]})
     else
         add(_r.targets,t)
     end
    end
    add(patterns[p.autovar].rects,_r)
    patterns[p.autovar].mins=p.mins
    patterns[p.autovar].maxs=p.maxs
   end
  end

 end
end

function updatepatterns()
 if dopatterns then
     if cp then
   cp.timer+=frz
         local rend=0
      for r in all(cp.rects) do
       local x=r.x
    local y=r.y
    local w=r.w
    local h=r.h
    local o=6
    local sh=min(r.sh,1)
    if cp.tpe==1 and r.sh<2 then
    if r.dx>0 then
     r.warnings={{x+w+o,y+h/2,x+w+o,lerp(y+h/2,y+1,sh)},{x+w+o,y+h/2,x+w+o,lerp(y+h/2,y+h-2,sh)}}
    elseif r.dx<0 then
     r.warnings={{x-o,y+h/2,x-o,lerp(y+h/2,y+1,sh)},{x-o,y+h/2,x-o,lerp(y+h/2,y+h-2,sh)}}
    end
    if r.dy>0 then
     r.warnings={{x+w/2,y+h+o,lerp(x+w/2,x+1,sh),y+h+o},{x+w/2,y+h+o,lerp(x+w/2,x+w-2,sh),y+h+o}}
    elseif r.dy<0 then
     r.warnings={{x+w/2,y-o,lerp(x+w/2,x+1,sh),y-o},{x+w/2,y-o,lerp(x+w/2,x+w-2,sh),y-o}}
    end
    end
       if r.sh==2 and not r.coldone then
        r.coldone=true
        for e in all(enemies) do
            if    e.x+e.s > r.x and
                         e.y+e.s > r.y and
                         e.x < r.x+r.w and
                         e.y < r.y+r.h and
                      e.p!=-1
                     then
                      e.isdying=true
                     end
                 end
       end
       if r.sh>=2 and r.shown then
           local t=r.targets[r.i]
           local spd=r.spd
           if t!=nil then
               if type(t)=="table" then
                if cp.mt then
                    x+=((t[1]-x)/spd)*frz
                    y+=((t[2]-y)/spd)*frz
                    w+=((t[3]-w)/spd)*frz
                    h+=((t[4]-h)/spd)*frz
                   else
                    if x>t[1] then
                     x-=spd*frz
                    elseif x<t[1] then
                     x+=spd*frz
                    end
                    if y>t[2] then
                     y-=spd*frz
                    elseif y<t[2] then
                     y+=spd*frz
                    end
                    if w>t[3] then
                     w-=spd*frz
                    elseif r.w<t[3] then
                     w+=spd*frz
                    end
                    if h>t[4] then
                     h-=spd*frz
                    elseif r.h<t[4] then
                     h+=spd*frz
                    end
                   end
                if round(x)==t[1]
                and round(y)==t[2]
                and round(w)==t[3]
                and round(h)==t[4]
                then
                 r.i+=1
                end
                r.x,r.y,r.w,r.h=x,y,w,h
               elseif type(t)=="number" then
                r.wait+=1*frz
                if r.wait>=t*60 then
                 r.wait=0
                 r.i+=1
                end
               elseif type(t)=="string" then
                   if t=="s" then
                    spawns=r.targets[r.i+1]
                    r.i+=2
                   elseif t=="fy" then
                    fy=r.targets[r.i+1]
                    r.i+=2
                   elseif t=="func" then
                    r.targets[r.i+1]()
                    r.i+=2
                   end
               end
              end
       else
        if r.shown then
         r.sh+=0.02*frz
         if (r.sh>1.99) r.sh=2 initspawns()
        else
         r.sh-=0.02*frz
         if (cp.spawns) spawns={}
         if (r.sh<0.05) r.sh=0
        end
       end

          if r.i>=#r.targets+1 then
           if cp.tpe==1 then
            r.finish=true
           else
            r.shown=false
           end
          end

       if not r.shown and r.sh<=0 then
        r.finish=true
       end
               
                -- move this up ?
          if r.finish then
        rend+=1
       end

      end
      if rend>=#cp.rects then
       local probs={}
       local counters={}
       for i,p in pairs(patterns) do
        add(probs,{i,p.prob})
        add(counters,{i,p.counter})
       end
       initpatterns()
       for i in all(probs) do
        patterns[i[1]].prob=i[2]
       end
       for i in all(counters) do
        patterns[i[1]].counter=i[2]
       end
       cp,fy=nil,true
       initspawns()
      end
        elseif not freeze then
         cpt+=1
         if cpt>=pt*60 then
          pt,cpt=10,0
          local pat=weightedtbl(currentrange())
          cp=pat[flr(rnd(#pat))+1]
          cp.counter+=1
       cp.prob=1
       for v in all(cp.var) do
        patterns[v].prob=1
       end
       if cp.spawns then
        spawns={}
       end
         end
        end
    end
end

function currentrange()
 local ps={}
 for p in all(patterns) do
  if score>=p.mins and score<=p.maxs then
   add(ps,p)
  end
 end
 return ps
end

function weightedtbl(tbl)
 local nt={}
 for i in all(tbl) do
  for w=1,i.prob do
   add(nt,i)
  end
 end
 return nt
end